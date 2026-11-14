// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/NumericalNote.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Container.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"

/**
 * @file TestConvDescriptorShards.cpp
 * @brief The shipped conv descriptors as the loader takes them: every per-arch shard of the
 *        production descriptor tree is loaded, its descriptor sets resolved, and each conv
 *        engine's state manager built, which is the step that drops a whole engine whose
 *        packs declare one kernel id twice on an arch (or otherwise fail validation), and
 *        each conv engine's numerical notes (RFC 0025) read. No device is used: every shard
 *        the build packed is checked, whatever the GPU. One device case reads the notes back
 *        from each engine's details, the field hipDNN serves as HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE.
 *
 * The per-kernel tests (TestGpuConv*) read the packs directly and match kernels one by one, so
 * an engine the loader drops would still pass them while serving nothing.
 */
namespace
{

using hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog;
using hipdnn_plugin_sdk::ingestor::makeStateManager;
using hipdnn_plugin_sdk::ingestor::resolveDescriptorSets;

/// The production descriptor tree this binary ships with: the plugin's search directory, or
/// the installed layout beside the binary.
std::filesystem::path productionRoot()
{
    std::vector<std::filesystem::path> roots;
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter unset(
            "HIPDNN_DESCRIPTOR_DIR", "");
        roots.push_back(hip_kernel_provider::kernel_ingestor_engine::descriptorSearchDirectory());
    }
    std::error_code exeError;
    for(auto dir = std::filesystem::read_symlink("/proc/self/exe", exeError).parent_path();
        !exeError && !dir.empty() && dir != dir.root_path();
        dir = dir.parent_path())
    {
        roots.push_back(dir / "lib" / "hipdnn_plugins" / "engines"
                        / HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR);
    }
    for(const auto& root : roots)
    {
        std::error_code error;
        if(std::filesystem::is_directory(root, error))
        {
            return root;
        }
    }
    return {};
}

/// The arch shards (gfx* folders) of @p root, sorted.
std::vector<std::filesystem::path> archShards(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> shards;
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(root, error))
    {
        std::error_code inner;
        if(entry.is_directory(inner) && entry.path().filename().string().rfind("gfx", 0) == 0)
        {
            shards.push_back(entry.path());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

TEST(TestConvDescriptorShards, EveryConvEngineValidatesOnEveryShippedShard)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto root = productionRoot();
    ASSERT_FALSE(root.empty()) << "no production descriptor tree beside this binary";
    const auto shards = archShards(root);
    ASSERT_FALSE(shards.empty()) << "no arch shard under " << root;
    for(const auto& shard : shards)
    {
        SCOPED_TRACE(shard.string());
        int convEngines = 0;
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            if(set.engine.name.rfind("hipkernel:CONV", 0) != 0)
            {
                continue;
            }
            SCOPED_TRACE(set.engine.name);
            ++convEngines;
            // A kernel id names one candidate per device: within a shard every pack the set
            // gathers reaches the shard's arch, so an id may appear once.
            std::map<std::string, int> declarations;
            for(const auto& pack : set.packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    const auto id = hipdnn_plugin_sdk::ingestor::toString(kernel.id);
                    EXPECT_EQ(++declarations[id], 1) << "kernel " << kernel.name << " (" << id
                                                     << ") declared again in pack " << pack.name;
                }
            }
            // The loader's own validation (which also refuses a repeated metadata tuple):
            // it drops the whole engine when this throws.
            EXPECT_NO_THROW(static_cast<void>(
                makeStateManager<Handle>(set, set.engine.graphMatchNativeSymbol)));
        }
        EXPECT_GT(convEngines, 0) << "no conv engine in the shard";
    }
}

/// The numerical notes a conv engine reports (RFC 0025): its algorithm class where cuDNN names
/// one (FFT, WINOGRAD), and never NONDETERMINISTIC. Every engine claims bitwise-reproducible
/// results, which each family's device test (TestGpuConv*) proves by launching every config it
/// runs twice (ConvDeterminismTestSupport.hpp).
std::vector<int32_t> expectedNumericalNotes(const std::string& engineName)
{
    if(engineName.rfind("hipkernel:CONV_FFT_", 0) == 0)
    {
        return {HIPDNN_NUMERICAL_NOTE_FFT};
    }
    if(engineName.rfind("hipkernel:CONV_WINOGRAD_", 0) == 0)
    {
        return {HIPDNN_NUMERICAL_NOTE_WINOGRAD};
    }
    return {};
}

TEST(TestConvDescriptorShards, EveryConvEngineDeclaresItsNumericalNotesOnEveryShippedShard)
{
    const auto root = productionRoot();
    ASSERT_FALSE(root.empty()) << "no production descriptor tree beside this binary";
    const auto shards = archShards(root);
    ASSERT_FALSE(shards.empty()) << "no arch shard under " << root;
    for(const auto& shard : shards)
    {
        SCOPED_TRACE(shard.string());
        int convEngines = 0;
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            if(set.engine.name.rfind("hipkernel:CONV", 0) != 0)
            {
                continue;
            }
            SCOPED_TRACE(set.engine.name);
            ++convEngines;
            // An absent list is no claim, which hipDNN reports as NONDETERMINISTIC.
            ASSERT_TRUE(set.engine.numericalNotes.has_value()) << "the UED has no numerical_notes";
            EXPECT_EQ(*set.engine.numericalNotes, expectedNumericalNotes(set.engine.name));
        }
        EXPECT_GT(convEngines, 0) << "no conv engine in the shard";
    }
}

TEST(TestGpuConvDescriptorNotes, EveryConvEngineReportsItsNumericalNotesInItsDetails)
{
    SKIP_IF_NO_DEVICES();

    namespace conv_test = hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
    hip_kernel_provider::core::Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;
    // Notes describe the engine, not the graph: any graph serves the query.
    const auto graph = conv_test::buildGraph(
        conv_test::specFor(conv_test::ConvShape{}, conv_test::ConvDirection::FORWARD));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapped(
        graph.GetBufferPointer(), graph.GetSize());
    int convEngines = 0;
    for(const auto& set : hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets())
    {
        if(set.engine.name.rfind("hipkernel:CONV", 0) != 0)
        {
            continue;
        }
        SCOPED_TRACE(set.engine.name);
        ++convEngines;
        hipdnnPluginConstData_t details{};
        engineManager.getEngineDetails(
            handle, wrapped, hipdnn_data_sdk::utilities::engineNameToId(set.engine.name), details);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(
            details.ptr, details.size);
        const auto notes = wrapper.numericalNotes();
        ASSERT_TRUE(notes.has_value()) << "the details carry no numerical_notes field";
        EXPECT_EQ(*notes, expectedNumericalNotes(set.engine.name));
    }
    EXPECT_GT(convEngines, 0) << "no conv engine loaded for this device";
}

} // namespace

#endif
