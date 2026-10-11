// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Adapted from dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/TestFlydslRmsNormPacks.cpp at commit 03911535bd0 (FlyDSL provider)

// Redundant with the CMake gate on purpose: if this file lands on an ungated source list
// it compiles to an empty translation unit instead of a suite reading an unstaged shard.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_TRITON)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/TritonRmsNormTestGraphs.hpp"

/**
 * @file TestTritonRmsNormPacks.cpp
 * @brief Census of what hipkernel:TritonRmsNorm loads from the staged descriptor shard.
 *
 * Part of hip_kernel_provider_census_tests, registered per shipped arch in
 * src/tests/CMakeLists.txt. Cases read the staged shard and never touch a device, so an
 * empty shard fails. The expected constants restate `triton/generators/_contract.py`.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// Per block_n: the launch threads (num_warps x 64 lanes) and the descriptor priority.
struct ExpectedBlock
{
    int64_t blockN;
    int64_t blockThreads;
    int64_t priority;
};

constexpr std::array<ExpectedBlock, 4> EXPECTED_BLOCKS{
    {{1024, 256, 400}, {2048, 256, 300}, {4096, 512, 200}, {8192, 512, 100}}};

constexpr std::array<const char*, 2> EXPECTED_DTYPES{"bf16", "f16"};

/// The symbol every instance exports.
constexpr const char* KERNEL_SYMBOL = "rmsnorm_kernel";

/// Dynamic LDS ceiling a launch can request without an opt-in.
constexpr int64_t MAX_SHARED_MEM_BYTES = 65536;

/// Every kernel of the loaded set with the architecture list it effectively carries: its
/// own when it declared one, otherwise its pack's.
struct LoadedKernel
{
    const KernelDescriptor* kernel;
    const std::vector<std::string>* arch;
};

std::vector<LoadedKernel> loadedKernels()
{
    std::vector<LoadedKernel> kernels;
    for(const auto& pack : loadedSet(TRITON_RMSNORM.engineName).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            kernels.push_back({&kernel, kernel.arch.empty() ? &pack.arch : &kernel.arch});
        }
    }
    return kernels;
}

TEST(TestTritonRmsNormPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(TRITON_RMSNORM.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, TRITON_RMSNORM.graphMatcher);

    // RMSNormAttributes is its own union arm, so no separate graph-scoped matcher exists.
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestTritonRmsNormPacks, ShipsFourBlockWidthsForBothDtypes)
{
    const auto kernels = loadedKernels();

    // Four block widths x two dtypes; also compared as (dtype, block_n) pairs so eight
    // copies of one instance fail.
    EXPECT_EQ(kernels.size(), 8U);

    std::set<std::pair<std::string, int64_t>> instances;
    for(const auto& [kernel, arch] : kernels)
    {
        const auto* dtype = tryGetMetadataField<std::string>(kernel->metadata, TRITON_DTYPE_FIELD);
        const auto* blockN = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_BLOCK_N_FIELD);
        ASSERT_NE(dtype, nullptr) << kernel->name << " declares no string dtype";
        ASSERT_NE(blockN, nullptr) << kernel->name << " declares no integer block_n";
        instances.emplace(*dtype, *blockN);

        // Pre-built code objects only: there is no source for the runtime compiler.
        EXPECT_EQ(kernel->source.kind, KernelSourceKind::KPACK) << kernel->name;
        EXPECT_FALSE(kernel->source.tocKey.empty()) << kernel->name;
        EXPECT_EQ(kernel->source.symbol, KERNEL_SYMBOL) << kernel->name;
        EXPECT_FALSE(kernel->source.sha256.empty()) << kernel->name;
    }

    std::set<std::pair<std::string, int64_t>> expected;
    for(const auto* dtype : EXPECTED_DTYPES)
    {
        for(const auto& block : EXPECTED_BLOCKS)
        {
            expected.emplace(dtype, block.blockN);
        }
    }
    EXPECT_EQ(instances, expected);
}

TEST(TestTritonRmsNormPacks, EveryKernelDeclaresTheSevenSlotSignature)
{
    // Explicit kernargs: x, scale, y, n_cols, eps, global_scratch, profile_scratch (the
    // last two are Triton's scratch slots, passed nullptr). Offsets are pinned too: the two
    // 4-byte by_value slots share one 8-byte word, so global_scratch is at 32, not 28.
    // Changing this table means changing launch() in TritonRmsNormNative.cpp.
    static const std::array<std::pair<const char*, uint32_t>, 7> s_slots{{{"global_buffer", 8},
                                                                          {"global_buffer", 8},
                                                                          {"global_buffer", 8},
                                                                          {"by_value", 4},
                                                                          {"by_value", 4},
                                                                          {"global_buffer", 8},
                                                                          {"global_buffer", 8}}};
    static const std::array<uint32_t, 7> s_offsets{0, 8, 16, 24, 28, 32, 40};

    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());
    for(const auto& [kernel, arch] : kernels)
    {
        const auto& signature = kernel->source.signature;
        ASSERT_GE(signature.size(), s_slots.size()) << kernel->name;
        for(size_t slot = 0; slot < s_slots.size(); ++slot)
        {
            const auto& argument = signature[slot];
            EXPECT_EQ(argument.kind, s_slots[slot].first) << kernel->name << " slot " << slot;
            EXPECT_EQ(argument.size, s_slots[slot].second) << kernel->name << " slot " << slot;
            EXPECT_EQ(argument.offset, s_offsets[slot]) << kernel->name << " slot " << slot;
        }

        // Anything after the explicit slots must be a hidden argument filled by the runtime.
        for(size_t slot = s_slots.size(); slot < signature.size(); ++slot)
        {
            EXPECT_EQ(signature[slot].kind.rfind("hidden_", 0), 0U)
                << kernel->name << " slot " << slot << " is '" << signature[slot].kind << "'";
        }
    }
}

TEST(TestTritonRmsNormPacks, EveryKernelDeclaresSharedAndBlockThreadsAndStaysUnder64KiB)
{
    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());

    std::map<int64_t, int64_t> threadsByBlock;
    for(const auto& block : EXPECTED_BLOCKS)
    {
        threadsByBlock[block.blockN] = block.blockThreads;
    }

    for(const auto& [kernel, arch] : kernels)
    {
        const auto* blockN = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_BLOCK_N_FIELD);
        const auto* threads
            = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_BLOCK_THREADS_FIELD);
        const auto* shared
            = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_SHARED_MEM_BYTES_FIELD);
        ASSERT_NE(blockN, nullptr) << kernel->name << " declares no integer block_n";
        ASSERT_NE(threads, nullptr) << kernel->name << " declares no integer block_threads";
        ASSERT_NE(shared, nullptr) << kernel->name << " declares no integer shared_mem_bytes";

        // block_threads is what the launch's block dimension is built from.
        const auto expectedThreads = threadsByBlock.find(*blockN);
        ASSERT_NE(expectedThreads, threadsByBlock.end())
            << kernel->name << " has an unexpected block_n " << *blockN;
        EXPECT_EQ(*threads, expectedThreads->second) << kernel->name;

        // Triton's LDS is entirely dynamic: a launch passing 0 computes wrong values
        // silently, and above 64 KiB a launch needs an opt-in the dispatch cannot express.
        EXPECT_GT(*shared, 0) << kernel->name;
        EXPECT_LE(*shared, MAX_SHARED_MEM_BYTES) << kernel->name;
    }
}

TEST(TestTritonRmsNormPacks, EveryKernelCoversTheStampedArch)
{
    // Supplied by the census entry, not detected from a device or the descriptors.
    const auto expectedArch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
    ASSERT_FALSE(expectedArch.empty())
        << "HIPDNN_TEST_EXPECTED_ARCH is unset, so this census has no expectation to check "
           "against";

    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());
    for(const auto& [kernel, arch] : kernels)
    {
        // Exactly this arch: a second one would name an arch the shard's object cannot run on.
        EXPECT_EQ(*arch, std::vector<std::string>{expectedArch}) << kernel->name;
    }
}

TEST(TestTritonRmsNormPacks, PrioritiesRankSmallerBlocksFirst)
{
    // Score is the descriptor priority, so this ladder makes the smallest admissible block
    // rank first.
    std::map<std::string, std::map<int64_t, int64_t>> priorityByDtypeAndBlock;
    for(const auto& [kernel, arch] : loadedKernels())
    {
        const auto* dtype = tryGetMetadataField<std::string>(kernel->metadata, TRITON_DTYPE_FIELD);
        const auto* blockN = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_BLOCK_N_FIELD);
        ASSERT_NE(dtype, nullptr) << kernel->name;
        ASSERT_NE(blockN, nullptr) << kernel->name;
        priorityByDtypeAndBlock[*dtype][*blockN] = kernel->priority;
    }

    ASSERT_EQ(priorityByDtypeAndBlock.size(), EXPECTED_DTYPES.size());
    for(const auto& [dtype, byBlock] : priorityByDtypeAndBlock)
    {
        ASSERT_EQ(byBlock.size(), EXPECTED_BLOCKS.size()) << dtype;

        // block_n ascends, so priorities must strictly descend.
        int64_t previous = 0;
        bool first = true;
        for(const auto& [blockN, priority] : byBlock)
        {
            if(!first)
            {
                EXPECT_LT(priority, previous)
                    << dtype << " block_n " << blockN << " must rank below the next smaller block";
            }
            previous = priority;
            first = false;
        }

        for(const auto& block : EXPECTED_BLOCKS)
        {
            EXPECT_EQ(byBlock.at(block.blockN), block.priority) << dtype << " " << block.blockN;
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
