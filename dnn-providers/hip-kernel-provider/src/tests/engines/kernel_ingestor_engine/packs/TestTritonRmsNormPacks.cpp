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
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/TritonRmsNormTestGraphs.hpp"

/**
 * @file TestTritonRmsNormPacks.cpp
 * @brief Census of what hipkernel:TritonRmsNorm loads from the staged descriptor shard,
 *        and the kernel matcher's per-instance boundaries generated from it.
 *
 * Part of hip_kernel_provider_census_tests, registered per shipped arch in
 * src/tests/CMakeLists.txt. Cases read the staged shard and never touch a device, so an
 * empty shard fails. The instance set itself (which dtypes and block widths ship) is the
 * generator's, triton/tritongen/ops/rmsnorm.py; these cases restate none of it and check
 * each staged kernel against its own metadata.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The symbol every instance exports.
constexpr const char* KERNEL_SYMBOL = "rmsnorm_kernel";

/// Dynamic LDS ceiling a launch can request without an opt-in.
constexpr int64_t MAX_SHARED_MEM_BYTES = 65536;

/// Lanes per wavefront on every arch the pack ships for, and HIP's threads-per-block cap.
constexpr int64_t WAVEFRONT_SIZE = 64;
constexpr int64_t MAX_BLOCK_THREADS = 1024;

/// HIP caps the threads in one grid dimension at UINT32_MAX.
constexpr int64_t MAX_GRID_THREADS = std::numeric_limits<uint32_t>::max();

/// Every kernel of the loaded set with its pack and the architecture list it effectively
/// carries: its own when it declared one, otherwise its pack's.
struct LoadedKernel
{
    const KernelDescriptorPack* pack;
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
            kernels.push_back({&pack, &kernel, kernel.arch.empty() ? &pack.arch : &kernel.arch});
        }
    }
    return kernels;
}

/// The definition the engine hands the kernel matcher for @p loaded.
KernelDefinition definitionOf(const LoadedKernel& loaded)
{
    return KernelDefinition{loaded.kernel->id,
                            loaded.pack->id,
                            loaded.pack->dispatchId,
                            loaded.kernel->source,
                            loaded.kernel->metadata,
                            loaded.kernel->priority,
                            *loaded.arch,
                            loaded.kernel->originDirectory,
                            loaded.kernel->name,
                            loaded.kernel->treeRoot};
}

/// @p kernel's integer metadata @p field. Fails the calling case when it is absent.
int64_t intField(const KernelDescriptor& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
    if(value == nullptr)
    {
        ADD_FAILURE() << kernel.name << " declares no integer " << field;
        return 0;
    }
    return *value;
}

/// The graph dtype a kernel's `dtype` metadata names, and the other built dtype.
struct DtypePair
{
    data_objects::DataType own;
    data_objects::DataType other;
};

std::optional<DtypePair> graphDtypesOf(const KernelDescriptor& kernel)
{
    const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, TRITON_DTYPE_FIELD);
    if(dtype != nullptr && *dtype == "bf16")
    {
        return DtypePair{data_objects::DataType::BFLOAT16, data_objects::DataType::HALF};
    }
    if(dtype != nullptr && *dtype == "f16")
    {
        return DtypePair{data_objects::DataType::HALF, data_objects::DataType::BFLOAT16};
    }
    ADD_FAILURE() << kernel.name << " declares no dtype this suite can build a graph for";
    return std::nullopt;
}

/// Whether the kernel matcher admits @p kernel for the graph @p options builds.
bool admits(const LoadedKernel& kernel, const TritonRmsNormGraphOptions& options)
{
    const GraphFixture fixture(buildTritonRmsNormGraph(options));
    const auto bound = bindingsFor(TRITON_RMSNORM, fixture.context());
    return matchesKernel(TRITON_RMSNORM, fixture.context(), definitionOf(kernel), bound);
}

/// A graph @p kernel serves apart from what a case changes: its own dtype, 8 rows of
/// exactly its block width.
TritonRmsNormGraphOptions servedGraph(const LoadedKernel& kernel)
{
    TritonRmsNormGraphOptions options;
    if(const auto dtypes = graphDtypesOf(*kernel.kernel); dtypes.has_value())
    {
        options.dataType = dtypes->own;
    }
    options.xDims = {8, intField(*kernel.kernel, TRITON_BLOCK_N_FIELD)};
    return options;
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
    for(const auto& [pack, kernel, arch] : kernels)
    {
        // Pre-built code objects only: there is no source for the runtime compiler.
        EXPECT_EQ(kernel->source.kind, KernelSourceKind::KPACK) << kernel->name;
        EXPECT_FALSE(kernel->source.tocKey.empty()) << kernel->name;
        EXPECT_EQ(kernel->source.symbol, KERNEL_SYMBOL) << kernel->name;
        EXPECT_FALSE(kernel->source.sha256.empty()) << kernel->name;

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

    for(const auto& [pack, kernel, arch] : kernels)
    {
        const auto* blockN = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_BLOCK_N_FIELD);
        const auto* threads
            = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_BLOCK_THREADS_FIELD);
        const auto* shared
            = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_SHARED_MEM_BYTES_FIELD);
        ASSERT_NE(blockN, nullptr) << kernel->name << " declares no integer block_n";
        ASSERT_NE(threads, nullptr) << kernel->name << " declares no integer block_threads";
        ASSERT_NE(shared, nullptr) << kernel->name << " declares no integer shared_mem_bytes";

        // block_threads is the launch's block dimension: whole wavefronts, within HIP's
        // per-block cap. block_n is the tile width every row must fit.
        EXPECT_GT(*blockN, 0) << kernel->name;
        EXPECT_GT(*threads, 0) << kernel->name;
        EXPECT_EQ(*threads % WAVEFRONT_SIZE, 0) << kernel->name;
        EXPECT_LE(*threads, MAX_BLOCK_THREADS) << kernel->name;

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
    for(const auto& [pack, kernel, arch] : kernels)
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
    for(const auto& [pack, kernel, arch] : loadedKernels())
    {
        const auto* dtype = tryGetMetadataField<std::string>(kernel->metadata, TRITON_DTYPE_FIELD);
        const auto* blockN = tryGetMetadataField<int64_t>(kernel->metadata, TRITON_BLOCK_N_FIELD);
        ASSERT_NE(dtype, nullptr) << kernel->name;
        ASSERT_NE(blockN, nullptr) << kernel->name;
        priorityByDtypeAndBlock[*dtype][*blockN] = kernel->priority;
    }
    ASSERT_FALSE(priorityByDtypeAndBlock.empty());

    for(const auto& [dtype, byBlock] : priorityByDtypeAndBlock)
    {
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
    }
}

// ---------------------------------------------------------------------------
// The kernel matcher at each staged kernel's own boundaries
// ---------------------------------------------------------------------------

TEST(TestTritonRmsNormPacks, EveryKernelAdmitsARowAsWideAsItsBlockAndRefusesOneWider)
{
    // One block_n-wide tile reduces a row, so it must cover the row.
    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());
    for(const auto& kernel : kernels)
    {
        auto options = servedGraph(kernel);
        EXPECT_TRUE(admits(kernel, options)) << kernel.kernel->name << " at N = block_n";

        options.xDims.back() += 1;
        EXPECT_FALSE(admits(kernel, options)) << kernel.kernel->name << " at N = block_n + 1";
    }
}

TEST(TestTritonRmsNormPacks, EveryKernelAdmitsTheLargestGridItsBlockCanLaunchAndNoMore)
{
    // One block_threads-wide block per row, and at most UINT32_MAX threads in the grid.
    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());
    for(const auto& kernel : kernels)
    {
        const auto threads = intField(*kernel.kernel, TRITON_BLOCK_THREADS_FIELD);
        ASSERT_GT(threads, 0) << kernel.kernel->name;

        auto options = servedGraph(kernel);
        options.xDims.front() = MAX_GRID_THREADS / threads;
        EXPECT_TRUE(admits(kernel, options))
            << kernel.kernel->name << " at " << options.xDims.front() << " rows";

        options.xDims.front() += 1;
        EXPECT_FALSE(admits(kernel, options))
            << kernel.kernel->name << " at " << options.xDims.front() << " rows";
    }
}

TEST(TestTritonRmsNormPacks, EveryKernelAdmitsItsOwnDtypeAndRefusesTheOther)
{
    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());
    for(const auto& kernel : kernels)
    {
        const auto dtypes = graphDtypesOf(*kernel.kernel);
        ASSERT_TRUE(dtypes.has_value());

        auto options = servedGraph(kernel);
        EXPECT_TRUE(admits(kernel, options)) << kernel.kernel->name;

        options.dataType = dtypes->other;
        EXPECT_FALSE(admits(kernel, options)) << kernel.kernel->name << " at the other dtype";
    }
}

TEST(TestTritonRmsNormPacks, EveryKernelAdmitsItsOwnAlignmentAndRefusesHalfOfItOnAnyOperand)
{
    // Each operand's declared alignment must be a positive multiple of the kernel's.
    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());
    for(const auto& kernel : kernels)
    {
        const auto alignment = intField(*kernel.kernel, TRITON_ALIGNMENT_FIELD);
        ASSERT_GT(alignment, 1) << kernel.kernel->name;

        auto options = servedGraph(kernel);
        options.xAlignment = alignment;
        options.scaleAlignment = alignment;
        options.yAlignment = alignment;
        EXPECT_TRUE(admits(kernel, options)) << kernel.kernel->name;

        for(auto* operand : {&options.xAlignment, &options.scaleAlignment, &options.yAlignment})
        {
            *operand = alignment / 2;
            EXPECT_FALSE(admits(kernel, options))
                << kernel.kernel->name << " with one operand at alignment " << alignment / 2;
            *operand = alignment;
        }
    }
}

TEST(TestTritonRmsNormPacks, NoKernelAdmitsARowWiderThanTheWidestBlock)
{
    // A row wider than every tile matches the graph but no kernel, so the engine declines.
    const auto kernels = loadedKernels();
    ASSERT_FALSE(kernels.empty());

    int64_t widest = 0;
    for(const auto& kernel : kernels)
    {
        widest = std::max(widest, intField(*kernel.kernel, TRITON_BLOCK_N_FIELD));
    }

    for(const auto& kernel : kernels)
    {
        auto options = servedGraph(kernel);
        options.xDims.back() = widest + 1;
        EXPECT_FALSE(admits(kernel, options)) << kernel.kernel->name;
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
