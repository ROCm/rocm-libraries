// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/NumericalNote.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestConvDescriptorShards.cpp
 * @brief The shipped conv descriptors as the loader takes them: every per-arch shard of the
 *        production descriptor tree is loaded, its descriptor sets resolved, and each conv
 *        engine's state manager built, which is the step that drops a whole engine whose
 *        packs declare one kernel id twice on an arch (or otherwise fail validation), and
 *        each conv engine's numerical notes (RFC 0025) read. No device is used: every shard
 *        the build packed is checked, whatever the GPU. What hipDNN serves from those notes
 *        (HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE) is checked by IntegrationConvCoverageCensus.
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

bool hasSuffix(const std::string& text, const std::string& suffix)
{
    return text.size() >= suffix.size()
           && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// The numerical notes a conv engine reports (RFC 0025): its algorithm class where cuDNN names
/// one (FFT; WINOGRAD with the tile note where one tile serves every config), TENSOR_CORE for the
/// engines of TF32 configs (cuDNN's note for TF32), and never NONDETERMINISTIC. Every engine
/// claims bitwise-reproducible results, which each family's device test (TestGpuConv*) proves
/// by launching every config it runs twice (ConvDeterminismTestSupport.hpp).
std::vector<int32_t> expectedNumericalNotes(const std::string& engineName)
{
    if(hasSuffix(engineName, "_TF32"))
    {
        return {HIPDNN_NUMERICAL_NOTE_TENSOR_CORE};
    }
    if(engineName.rfind("hipkernel:CONV_FFT_", 0) == 0)
    {
        return {HIPDNN_NUMERICAL_NOTE_FFT};
    }
    if(engineName == "hipkernel:CONV_WINOGRAD_WRW")
    {
        return {HIPDNN_NUMERICAL_NOTE_WINOGRAD};
    }
    if(hasSuffix(engineName, "_F43"))
    {
        return {HIPDNN_NUMERICAL_NOTE_WINOGRAD, HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6};
    }
    if(engineName.rfind("hipkernel:CONV_WINOGRAD_", 0) == 0)
    {
        return {HIPDNN_NUMERICAL_NOTE_WINOGRAD, HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4};
    }
    return {};
}

template <typename T>
const T* metadataOf(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel,
                    const std::string& field)
{
    const auto it = kernel.metadata.find(field);
    return it == kernel.metadata.end() ? nullptr : std::get_if<T>(&it->second);
}

/// A TF32 config: fp32 through rounded operands (three-bf16: math 1 or "bf16x3"; xf32 atoms).
bool isTf32Config(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel)
{
    const auto* math = metadataOf<int64_t>(kernel, "math");
    const auto* mathName = metadataOf<std::string>(kernel, "math");
    const auto* atom = metadataOf<std::string>(kernel, "atom");
    return (math != nullptr && *math == 1) || (mathName != nullptr && *mathName == "bf16x3")
           || (atom != nullptr && hasSuffix(*atom, "_xf32"));
}

/// Which configs an engine may list: a *_TF32 engine only TF32 configs and every other engine
/// none; a Winograd forward / backward-data engine its own kind (BF16: bf16, F43: F(4,3), the
/// exact-tile engine neither).
bool engineListsConfig(const std::string& engineName,
                       const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel)
{
    if(isTf32Config(kernel) != hasSuffix(engineName, "_TF32"))
    {
        return false;
    }
    if(engineName.rfind("hipkernel:CONV_WINOGRAD_", 0) != 0
       || engineName == "hipkernel:CONV_WINOGRAD_WRW")
    {
        return true;
    }
    const auto* dtype = metadataOf<std::string>(kernel, "dtype");
    const auto* m = metadataOf<int64_t>(kernel, "m");
    const bool bf16 = dtype != nullptr && *dtype == "BFLOAT16";
    const bool f43 = m != nullptr && *m == 4;
    return bf16 == hasSuffix(engineName, "_BF16") && f43 == hasSuffix(engineName, "_F43");
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

// The configs a note marks sit only in the engines that report it: the exact engines list no
// TF32, F(4,3) or bf16 Winograd forward / backward-data config, and each sibling engine lists
// only its own kind.
TEST(TestConvDescriptorShards, EachConvEngineListsOnlyTheConfigsItsNotesDescribe)
{
    const auto root = productionRoot();
    ASSERT_FALSE(root.empty()) << "no production descriptor tree beside this binary";
    for(const auto& shard : archShards(root))
    {
        SCOPED_TRACE(shard.string());
        std::set<std::string> names;
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            if(set.engine.name.rfind("hipkernel:CONV", 0) != 0)
            {
                continue;
            }
            names.insert(set.engine.name);
            size_t kernels = 0;
            for(const auto& pack : set.packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    ++kernels;
                    EXPECT_TRUE(engineListsConfig(set.engine.name, kernel))
                        << set.engine.name << " lists " << kernel.name;
                }
            }
            EXPECT_GT(kernels, 0U) << set.engine.name << " lists no config";
        }
        // Every family with TF32 configs ships its TF32 engine beside the exact one, and
        // Winograd its bf16 and F(4,3) engines.
        const std::vector<std::pair<std::string, std::string>> siblings = {
            {"hipkernel:CONV_IGEMM_FWD", "hipkernel:CONV_IGEMM_FWD_TF32"},
            {"hipkernel:CONV_IGEMM_BWD", "hipkernel:CONV_IGEMM_BWD_TF32"},
            {"hipkernel:CONV_DIRECT_FWD", "hipkernel:CONV_DIRECT_FWD_TF32"},
            {"hipkernel:CONV_DIRECT_BWD", "hipkernel:CONV_DIRECT_BWD_TF32"},
            {"hipkernel:CONV_WRW", "hipkernel:CONV_WRW_TF32"},
            {"hipkernel:CONV_IGEMM_NCHW_BWD", "hipkernel:CONV_IGEMM_NCHW_BWD_TF32"},
            {"hipkernel:CONV_GROUP_FWD", "hipkernel:CONV_GROUP_FWD_TF32"},
            {"hipkernel:CONV_GROUP_BWD", "hipkernel:CONV_GROUP_BWD_TF32"},
            {"hipkernel:CONV_WINOGRAD_FWD", "hipkernel:CONV_WINOGRAD_FWD_BF16"},
            {"hipkernel:CONV_WINOGRAD_BWD", "hipkernel:CONV_WINOGRAD_BWD_BF16"},
            {"hipkernel:CONV_WINOGRAD_FWD", "hipkernel:CONV_WINOGRAD_FWD_F43"},
            {"hipkernel:CONV_WINOGRAD_BWD", "hipkernel:CONV_WINOGRAD_BWD_F43"},
        };
        for(const auto& [exact, sibling] : siblings)
        {
            EXPECT_EQ(names.count(exact), names.count(sibling)) << exact << " / " << sibling;
        }
    }
}

} // namespace

#endif
