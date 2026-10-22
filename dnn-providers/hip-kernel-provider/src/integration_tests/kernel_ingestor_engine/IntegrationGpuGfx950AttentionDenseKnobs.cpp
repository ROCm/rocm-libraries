// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuGfx950AttentionDenseKnobs.cpp
 * @brief The block_m/block_n knobs of hipkernel:Gfx950AttentionDense, end to end through
 *        the frontend on a gfx950 device: a forced tile selects exactly its kernel, that
 *        kernel launches with its own geometry and computes what the CPU reference
 *        computes, and a knob pair no kernel has is refused before any plan exists.
 *
 * Each expected kernel id is resolved at run time from the production descriptors by
 * expectedKernelId(). Every forced tile except the 256/64 baseline is paired with a shape
 * whose no-knob winner is a different tile, so a knob that failed to reach the plugin would
 * select that winner and fail the id check. The baseline always wins where it fits, so its
 * two cases select the no-knob winner by design. Those no-knob winners are the native
 * scorer's; where a shipped UHD model ranks the engine, a cold case accepts the model's
 * winner as long as it serves the shape, and a forced case still filters to one kernel.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr ServedEngine ENGINE{"hipkernel:Gfx950AttentionDense", "gfx950"};

constexpr const char* BLOCK_M_KNOB = "block_m";
constexpr const char* BLOCK_N_KNOB = "block_n";

// The knob case that the save and load cases build. Its graph sets the scale 1/sqrt(64), so a
// loaded plan that applies the default scale 1.0 fails the comparison.
constexpr const char* SAVE_LOAD_CASE = "D64_Bm128Bn64";

using Mask = SdpaMask;
using GraphShape = IngestorSdpaGraphSpec;

struct Tile
{
    int64_t blockM;
    int64_t blockN;
};

/// One graph and the tile the engine must serve it with. A forced case sets block_m and
/// block_n to the tile; a cold case sets no knob, and the tile is the one the engine's own
/// ranking picks.
struct KnobCase
{
    const char* name;
    GraphShape shape;
    bool forced;
    Tile tile;
};

/// One graph and a knob pair whose values the engine advertises for it one by one while
/// no kernel it admits carries both.
struct UnsatisfiableCase
{
    const char* name;
    GraphShape shape;
    Tile forcedTile;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const KnobCase& knobCase, std::ostream* os)
{
    *os << knobCase.name;
}

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const UnsatisfiableCase& unsatisfiableCase, std::ostream* os)
{
    *os << unsatisfiableCase.name;
}

constexpr GraphShape makeShape(DataType dataType,
                               int64_t headSize,
                               int64_t queryHeads,
                               int64_t kvHeads,
                               Mask mask,
                               DataType mmaCoreMode,
                               int64_t batch,
                               int64_t seqQ,
                               int64_t seqKv)
{
    return {dataType, headSize, queryHeads, kvHeads, mask, batch, seqQ, seqKv, 0, mmaCoreMode};
}

GraphShape withoutScale(GraphShape shape)
{
    shape.omitScale = true;
    return shape;
}

GraphShape withScale(GraphShape shape, float scale)
{
    shape.scaleOverride = scale;
    return shape;
}

/// Forced (tile, head size) pairs and cold cases first; then one case for each remaining
/// catalog head configuration, in ascending (head size, Hq, Hkv) order at B=1, Sq=Skv=256;
/// then further cases until every (head size, dtype, causal, tile) class the catalog compiles
/// runs at least once; then cold cases of an fp16 graph with an explicit mma_core_mode; then
/// two forced cases with no attn_scale_value and one with a non-default attn_scale_value.
/// Within them: bounds on both corners and both deprecated flags, fp16 and bf16, MHA, GQA and
/// MQA, and forced cases with B > 1 and Sq != Skv.
std::vector<KnobCase> knobCases()
{
    constexpr auto FP16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    constexpr auto MMA_UNSET = DataType::NOT_SET;
    constexpr bool FORCED = true;
    constexpr bool COLD = false;
    return {
        // D64: all seven legal tiles.
        {"D64_Bm128Bn32",
         makeShape(BF16, 64, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 32}},
        {"D64_Bm128Bn64",
         makeShape(FP16, 64, 16, 2, Mask::BOUNDS_TOP_LEFT, MMA_UNSET, 2, 128, 256),
         FORCED,
         Tile{128, 64}},
        {"D64_Bm128Bn128",
         makeShape(BF16, 64, 8, 1, Mask::NO_MASK, MMA_UNSET, 2, 384, 128),
         FORCED,
         Tile{128, 128}},
        {"D64_Bm256Bn32",
         makeShape(FP16, 64, 8, 8, Mask::CAUSAL_MASK_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 32}},
        {"D64_Bm256Bn128",
         makeShape(BF16, 64, 16, 16, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        {"D64_Bm256Bn256",
         makeShape(FP16, 64, 12, 12, Mask::NO_MASK, MMA_UNSET, 1, 512, 256),
         FORCED,
         Tile{256, 256}},
        // The baseline tile: forced and cold select the same kernel.
        {"D64_Bm256Bn64",
         makeShape(BF16, 64, 10, 10, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 64}},
        // D128: all six legal tiles.
        {"D128_Bm128Bn32",
         makeShape(FP16, 128, 8, 2, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 32}},
        {"D128_Bm128Bn64",
         makeShape(BF16, 128, 8, 1, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 1, 128, 128),
         FORCED,
         Tile{128, 64}},
        {"D128_Bm128Bn128",
         makeShape(FP16, 128, 4, 4, Mask::NO_MASK, MMA_UNSET, 3, 128, 384),
         FORCED,
         Tile{128, 128}},
        {"D128_Bm256Bn32",
         makeShape(BF16, 128, 16, 2, Mask::NO_MASK, MMA_UNSET, 2, 256, 480),
         FORCED,
         Tile{256, 32}},
        {"D128_Bm256Bn128",
         makeShape(BF16, 128, 9, 9, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        // The baseline tile: forced and cold select the same kernel.
        {"D128_Bm256Bn64",
         makeShape(FP16, 128, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 64}},
        {"Cold_MmaHalf",
         makeShape(BF16, 64, 16, 2, Mask::BOUNDS_TOP_LEFT, FP16, 2, 256, 512),
         COLD,
         Tile{256, 64}},
        {"Cold_MmaBfloat16",
         makeShape(BF16, 128, 8, 1, Mask::NO_MASK, BF16, 3, 384, 256),
         COLD,
         Tile{128, 32}},
        {"Cold_CausalBottomRightFlag",
         makeShape(FP16, 64, 8, 8, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 2, 512, 512),
         COLD,
         Tile{256, 64}},
        {"Cold_D128Heads9",
         makeShape(BF16, 128, 9, 9, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 2, 256, 256),
         COLD,
         Tile{256, 64}},
        // One case for each catalog head configuration the cases above leave out, at
        // Sq=Skv=256 where the cold winner is 256/64, so no forced tile is the cold winner.
        // Dtype, mask and tile are chosen, not rotated: together with the rows around them
        // they run every (head size, dtype, causal, tile) class the catalog compiles.
        {"D64_Hq20Kv20_Bm128Bn64",
         makeShape(BF16, 64, 20, 20, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 64}},
        {"D64_Hq32Kv4_Bm256Bn32",
         makeShape(BF16, 64, 32, 4, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 32}},
        {"D64_Hq32Kv8_Bm256Bn128",
         makeShape(BF16, 64, 32, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        {"D64_Hq32Kv32_Bm256Bn256",
         makeShape(BF16, 64, 32, 32, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 256}},
        {"D64_Hq64Kv8_Bm128Bn32",
         makeShape(BF16, 64, 64, 8, Mask::BOUNDS_TOP_LEFT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 32}},
        {"D64_Hq128Kv128_Bm128Bn64",
         makeShape(BF16, 64, 128, 128, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 64}},
        {"D128_Hq16Kv16_Bm128Bn64",
         makeShape(BF16, 128, 16, 16, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 64}},
        {"D128_Hq24Kv24_Bm128Bn128",
         makeShape(BF16, 128, 24, 24, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 128}},
        {"D128_Hq28Kv4_Bm256Bn128",
         makeShape(BF16, 128, 28, 4, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        {"D128_Hq32Kv4_Bm128Bn32",
         makeShape(BF16, 128, 32, 4, Mask::CAUSAL_MASK_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 32}},
        {"D128_Hq32Kv8_Bm128Bn128",
         makeShape(BF16, 128, 32, 8, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 128}},
        {"D128_Hq32Kv32_Bm256Bn32",
         makeShape(BF16, 128, 32, 32, Mask::BOUNDS_TOP_LEFT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 32}},
        {"D128_Hq40Kv8_Bm128Bn32",
         makeShape(FP16, 128, 40, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 32}},
        {"D128_Hq40Kv10_Bm128Bn64",
         makeShape(FP16, 128, 40, 10, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 64}},
        {"D128_Hq40Kv40_Bm256Bn32",
         makeShape(FP16, 128, 40, 40, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 32}},
        {"D128_Hq48Kv8_Bm256Bn128",
         makeShape(FP16, 128, 48, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        {"D128_Hq64Kv4_Bm128Bn64",
         makeShape(FP16, 128, 64, 4, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 64}},
        {"D128_Hq64Kv8_Bm128Bn128",
         makeShape(FP16, 128, 64, 8, Mask::CAUSAL_MASK_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 128}},
        {"D128_Hq64Kv64_Bm256Bn32",
         makeShape(FP16, 128, 64, 64, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 32}},
        {"D128_Hq96Kv8_Bm256Bn128",
         makeShape(FP16, 128, 96, 8, Mask::BOUNDS_TOP_LEFT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        {"D128_Hq128Kv8_Bm128Bn32",
         makeShape(FP16, 128, 128, 8, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 32}},
        // The remaining D64 classes: there are more D64 classes than D64 head configurations.
        {"D64_Hq8Kv1_Bm128Bn128",
         makeShape(BF16, 64, 8, 1, Mask::CAUSAL_MASK_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 128}},
        {"D64_Hq8Kv8_Bm256Bn32",
         makeShape(BF16, 64, 8, 8, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 32}},
        {"D64_Hq10Kv10_Bm256Bn256",
         makeShape(BF16, 64, 10, 10, Mask::BOUNDS_TOP_LEFT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 256}},
        {"D64_Hq12Kv12_Bm128Bn64",
         makeShape(FP16, 64, 12, 12, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 64}},
        {"D64_Hq16Kv2_Bm128Bn128",
         makeShape(FP16, 64, 16, 2, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 128}},
        {"D64_Hq16Kv16_Bm256Bn32",
         makeShape(FP16, 64, 16, 16, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 32}},
        {"D64_Hq20Kv20_Bm256Bn128",
         makeShape(FP16, 64, 20, 20, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        {"D64_Hq8Kv1_Bm128Bn32",
         makeShape(FP16, 64, 8, 1, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 32}},
        {"D64_Hq8Kv8_Bm128Bn128",
         makeShape(FP16, 64, 8, 8, Mask::CAUSAL_MASK_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{128, 128}},
        {"D64_Hq10Kv10_Bm256Bn128",
         makeShape(FP16, 64, 10, 10, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 128}},
        {"D64_Hq12Kv12_Bm256Bn256",
         makeShape(FP16, 64, 12, 12, Mask::BOUNDS_TOP_LEFT, MMA_UNSET, 1, 256, 256),
         FORCED,
         Tile{256, 256}},
        // The non-causal 256/64 classes. 256/64 is the cold winner at Sq=Skv=256, so only a
        // cold case reaches it here.
        {"Cold_D64Hq8Kv8_Bf16NoMask",
         makeShape(BF16, 64, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         COLD,
         Tile{256, 64}},
        {"Cold_D64Hq16Kv16_Fp16NoMask",
         makeShape(FP16, 64, 16, 16, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         COLD,
         Tile{256, 64}},
        {"Cold_D128Hq8Kv8_Bf16NoMask",
         makeShape(BF16, 128, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         COLD,
         Tile{256, 64}},
        // An fp16 graph with each explicit mma_core_mode.
        {"Cold_Fp16MmaHalf",
         makeShape(FP16, 128, 16, 16, Mask::BOUNDS_TOP_LEFT, FP16, 2, 256, 512),
         COLD,
         Tile{256, 64}},
        {"Cold_Fp16MmaBfloat16",
         makeShape(FP16, 64, 32, 8, Mask::NO_MASK, BF16, 1, 384, 256),
         COLD,
         Tile{128, 32}},
        // No attn_scale_value: the CPU reference applies 1.0 (no scaling), so a kernel
        // launched with any other scale fails the comparison.
        {"D64_Bm256Bn64_NoScale",
         withoutScale(
             makeShape(BF16, 64, 10, 10, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256)),
         FORCED,
         Tile{256, 64}},
        {"D128_Bm256Bn64_NoScale",
         withoutScale(makeShape(FP16, 128, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256)),
         FORCED,
         Tile{256, 64}},
        // A baked scale other than 1/sqrt(D) and 1.0: the engine must launch with the
        // graph's value, not a default.
        {"D128_Bm256Bn64_Scale2",
         withScale(makeShape(FP16, 128, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256), 2.0F),
         FORCED,
         Tile{256, 64}},
    };
}

/// The D64 tiles this graph admits are 128/32, 128/64, 128/128, 256/32, 256/64, 256/128
/// and 256/256, so block_m=128 and block_n=256 are each advertised while no D64 kernel
/// is 128/256.
std::vector<UnsatisfiableCase> unsatisfiableCases()
{
    return {
        {"D64_Bm128Bn256",
         makeShape(DataType::HALF, 64, 8, 8, Mask::NO_MASK, DataType::NOT_SET, 1, 256, 256),
         Tile{128, 256}},
    };
}

std::vector<KnobSetting> knobSettingsFor(const Tile& tile)
{
    std::vector<KnobSetting> settings;
    settings.emplace_back(BLOCK_M_KNOB, tile.blockM);
    settings.emplace_back(BLOCK_N_KNOB, tile.blockN);
    return settings;
}

/// The id of the one kernel this engine's gfx950 packs carry for @p shape's semantic fields
/// and @p tile, as the selection line spells it, or nullopt (with a recorded failure) when
/// that is not exactly one.
std::optional<std::string> expectedKernelId(const GraphShape& shape, const Tile& tile)
{
    const detail::MetadataKey key{
        {"dtype", detail::catalogDtypeName(shape.dataType)},
        {"head_size", shape.headSize},
        {"num_query_heads", shape.queryHeads},
        {"num_kv_heads", shape.kvHeads},
        {"causal", detail::catalogCausal(shape.mask)},
        {"ragged", int64_t{0}},
        {"sliding_window", int64_t{0}},
        {"block_m", tile.blockM},
        {"block_n", tile.blockN},
    };
    return detail::onlyServedKernelCarrying(ENGINE, key);
}

/// The ids of every kernel this engine's gfx950 packs carry for @p shape's semantic fields,
/// at any tile: the kernels that can serve the shape.
std::set<std::string> servingKernelIds(const GraphShape& shape)
{
    const detail::MetadataKey key{
        {"dtype", detail::catalogDtypeName(shape.dataType)},
        {"head_size", shape.headSize},
        {"num_query_heads", shape.queryHeads},
        {"num_kv_heads", shape.kvHeads},
        {"causal", detail::catalogCausal(shape.mask)},
        {"ragged", int64_t{0}},
        {"sliding_window", int64_t{0}},
    };

    std::set<std::string> ids;
    detail::forEachServedKernel(
        ENGINE, [&key, &ids](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            if(detail::carriesKey(kernel, key))
            {
                ids.insert(hipdnn_plugin_sdk::ingestor::toString(kernel.id));
            }
        });
    return ids;
}

/// Whether a shipped UHD model ranked this engine's candidates, as UhdKernelHeuristic's
/// selection trace reports it. Without one the engine keeps its native-scored order.
bool rankedByModel(const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder)
{
    const std::string engine = std::string("engine '") + ENGINE.name + "'";
    for(const auto& log : recorder.getRecordedLogs())
    {
        if(log.message.find("uhd trace: ") != std::string::npos
           && log.message.find(engine) != std::string::npos
           && log.message.find(" decided_by=model ") != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

} // namespace

template <typename TestCaseType>
class IntegrationGpuGfx950AttentionDenseBase : public ServedEngineHarness<TestCaseType>
{
protected:
    IntegrationGpuGfx950AttentionDenseBase()
        : ServedEngineHarness<TestCaseType>(ENGINE, "gfx950-dense-knobs-case")
    {
    }
};

using IntegrationGpuGfx950AttentionDenseKnobs = IntegrationGpuGfx950AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx950AttentionDenseKnobFilter
    = IntegrationGpuGfx950AttentionDenseBase<UnsatisfiableCase>;

using IntegrationGpuGfx950AttentionDenseKnobCoverage
    = IntegrationGpuGfx950AttentionDenseBase<KnobCase>;

/// knobCases() is a static table because its cases register before main() names the
/// descriptor root. This keeps it honest: the (head_size, Hq, Hkv) configurations it covers
/// must be exactly those the shipped catalog carries.
TEST_F(IntegrationGpuGfx950AttentionDenseKnobCoverage, CoversEveryCatalogHeadConfiguration)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
        << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production descriptors to "
           "read the catalog's head configurations from";

    using HeadConfig = std::tuple<int64_t, int64_t, int64_t>;

    std::set<HeadConfig> tableConfigs;
    for(const auto& knobCase : knobCases())
    {
        tableConfigs.emplace(
            knobCase.shape.headSize, knobCase.shape.queryHeads, knobCase.shape.kvHeads);
    }

    std::set<HeadConfig> catalogConfigs;
    detail::forEachServedKernel(
        ENGINE, [&catalogConfigs](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            const auto intField = [&kernel](const char* field) -> const int64_t* {
                const auto value = kernel.metadata.find(field);
                return value == kernel.metadata.end() ? nullptr
                                                      : std::get_if<int64_t>(&value->second);
            };
            const auto* headSize = intField("head_size");
            const auto* queryHeads = intField("num_query_heads");
            const auto* kvHeads = intField("num_kv_heads");
            if(headSize == nullptr || queryHeads == nullptr || kvHeads == nullptr)
            {
                ADD_FAILURE() << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                              << " carries no integer head_size, num_query_heads and num_kv_heads";
                return;
            }
            catalogConfigs.emplace(*headSize, *queryHeads, *kvHeads);
        });

    const auto onlyIn = [](const std::set<HeadConfig>& from, const std::set<HeadConfig>& other) {
        std::string text;
        for(const auto& [headSize, queryHeads, kvHeads] : from)
        {
            if(other.count({headSize, queryHeads, kvHeads}) == 0)
            {
                text += " (D" + std::to_string(headSize) + ", Hq" + std::to_string(queryHeads)
                        + ", Hkv" + std::to_string(kvHeads) + ")";
            }
        }
        return text.empty() ? std::string(" none") : text;
    };
    EXPECT_EQ(tableConfigs, catalogConfigs)
        << "in the catalog but not knobCases():" << onlyIn(catalogConfigs, tableConfigs)
        << "\nin knobCases() but not the catalog:" << onlyIn(tableConfigs, catalogConfigs)
        << "\nunder HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
        << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";
}

/// Every row runs one compiled kernel, and what the compiler specializes on is the head size,
/// dtype, causal flag and tile. The table must run every such class the catalog carries, so a
/// row edited or removed cannot silently leave a code path unexecuted.
TEST_F(IntegrationGpuGfx950AttentionDenseKnobCoverage, CoversEveryCatalogKernelClass)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
        << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production descriptors to "
           "read the catalog's kernel classes from";

    using KernelClass = std::tuple<int64_t, std::string, int64_t, int64_t, int64_t>;

    std::set<KernelClass> tableClasses;
    for(const auto& knobCase : knobCases())
    {
        tableClasses.emplace(knobCase.shape.headSize,
                             detail::catalogDtypeName(knobCase.shape.dataType),
                             detail::catalogCausal(knobCase.shape.mask),
                             knobCase.tile.blockM,
                             knobCase.tile.blockN);
    }

    std::set<KernelClass> catalogClasses;
    detail::forEachServedKernel(
        ENGINE, [&catalogClasses](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            const auto intField = [&kernel](const char* field) -> const int64_t* {
                const auto value = kernel.metadata.find(field);
                return value == kernel.metadata.end() ? nullptr
                                                      : std::get_if<int64_t>(&value->second);
            };
            const auto dtype = kernel.metadata.find("dtype");
            const auto* dtypeName = dtype == kernel.metadata.end()
                                        ? nullptr
                                        : std::get_if<std::string>(&dtype->second);
            const auto* headSize = intField("head_size");
            const auto* causal = intField("causal");
            const auto* blockM = intField("block_m");
            const auto* blockN = intField("block_n");
            if(dtypeName == nullptr || headSize == nullptr || causal == nullptr || blockM == nullptr
               || blockN == nullptr)
            {
                ADD_FAILURE() << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                              << " carries no string dtype and integer head_size, causal, block_m "
                                 "and block_n";
                return;
            }
            catalogClasses.emplace(*headSize, *dtypeName, *causal, *blockM, *blockN);
        });

    const auto onlyIn = [](const std::set<KernelClass>& from, const std::set<KernelClass>& other) {
        std::string text;
        for(const auto& kernelClass : from)
        {
            if(other.count(kernelClass) == 0)
            {
                const auto& [headSize, dtype, causal, blockM, blockN] = kernelClass;
                text += " (D" + std::to_string(headSize) + ", " + dtype + ", causal "
                        + std::to_string(causal) + ", " + std::to_string(blockM) + "/"
                        + std::to_string(blockN) + ")";
            }
        }
        return text.empty() ? std::string(" none") : text;
    };
    EXPECT_EQ(tableClasses, catalogClasses)
        << "in the catalog but not knobCases():" << onlyIn(catalogClasses, tableClasses)
        << "\nin knobCases() but not the catalog:" << onlyIn(tableClasses, catalogClasses)
        << "\nunder HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
        << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";
}

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter reads as the cold winner; the
/// output is then checked against the CPU reference, which catches a kernel launched
/// with any geometry but its own. A cold case expects the native scorer's tile only when
/// the engine ranks natively: with a shipped UHD model ranking, the model picks the cold
/// winner, and the case checks that winner can serve the shape instead.
TEST_P(IntegrationGpuGfx950AttentionDenseKnobs, SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const detail::SelectedKernelReport report(ENGINE, testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = detail::buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    const auto settings
        = testCase.forced ? knobSettingsFor(testCase.tile) : std::vector<KnobSetting>{};
    auto result = sdpa.graph->create_execution_plan_ext(engineId(), settings);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = detail::selectedKernelId(ENGINE, recorder);
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << ENGINE.name << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    std::string expectedId;
    if(!testCase.forced && rankedByModel(recorder))
    {
        // A model's choice is any kernel that serves this shape; pinning its id would tie
        // the test to one training run.
        const auto serving = servingKernelIds(testCase.shape);
        EXPECT_EQ(serving.count(*selected), 1U)
            << "the model selected " << *selected << ", which does not serve this shape. "
            << "Captured logs:\n"
            << recorder.getRecordedLogsAsString();
        expectedId = *selected;
    }
    else
    {
        // Resolved after the plan is built, so a lookup failure still leaves the selected
        // kernel in the KNOB_SELECTED line.
        const auto tileKernelId = expectedKernelId(testCase.shape, testCase.tile);
        ASSERT_TRUE(tileKernelId.has_value());
        EXPECT_EQ(*selected, *tileKernelId) << "Captured logs:\n"
                                            << recorder.getRecordedLogsAsString();
        expectedId = *tileKernelId;
    }

    // Rank 0: a cold case serves the heuristic's front rather than a fallback past a
    // kernel that failed to load, and a forced case filters to its one kernel.
    auto selectionLine = std::string(detail::SELECTED_KERNEL_LOG) + expectedId + " at rank 0";
    if(testCase.forced)
    {
        selectionLine += " from 1 candidate(s)";
    }
    EXPECT_TRUE(recorder.hasLogContaining(selectionLine))
        << "expected '" << selectionLine << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = detail::sdpaForwardTolerance(testCase.shape.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    verifyBuiltGraph(context, /*seed=*/0);
}

/// The frontend validates each knob value on its own against the values the engine
/// advertises for this graph, never the pair. A pair of advertised values that no kernel
/// carries therefore reaches the plugin's knob filter, whose refusal is the only thing
/// that stands between it and a plan built from nothing.
TEST_P(IntegrationGpuGfx950AttentionDenseKnobFilter, RefusesAKnobPairNoKernelCarriesAndBuildsNoPlan)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const auto& recorder = capture.recorder();

    auto sdpa = detail::buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    std::vector<Knob> knobs;
    auto result = sdpa.graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_TRUE(detail::advertises(knobs, BLOCK_M_KNOB, testCase.forcedTile.blockM))
        << BLOCK_M_KNOB << "=" << testCase.forcedTile.blockM
        << " is not advertised for this graph, so the frontend refuses it before the plugin";
    ASSERT_TRUE(detail::advertises(knobs, BLOCK_N_KNOB, testCase.forcedTile.blockN))
        << BLOCK_N_KNOB << "=" << testCase.forcedTile.blockN
        << " is not advertised for this graph, so the frontend refuses it before the plugin";

    result
        = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase.forcedTile));
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR) << result.err_msg;

    const std::string refusal
        = std::string("engine '") + ENGINE.name + "' has no kernel satisfying";
    EXPECT_NE(result.err_msg.find(refusal), std::string::npos)
        << "expected '" << refusal << "' in: " << result.err_msg;

    // No plan: nothing was compiled for build_plans() to finalize, no engine backs a plan,
    // and the plugin never selected a kernel.
    result = sdpa.graph->build_plans();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;

    int64_t planEngineId = 0;
    EXPECT_NE(sdpa.graph->get_execution_plan_engine_id(planEngineId).code, ErrorCode::OK)
        << "a plan backed by engine " << planEngineId << " exists";

    EXPECT_FALSE(detail::selectedKernelId(ENGINE, recorder).has_value())
        << "Captured logs:\n"
        << recorder.getRecordedLogsAsString();
}

class IntegrationGpuGfx950AttentionDenseSaveLoad
    : public IntegrationGpuGfx950AttentionDenseBase<KnobCase>
{
protected:
    // Builds the forced-tile graph of the save and load knob case, validates one run and saves
    // the plan in form. Then it destroys the graph, loads the plan with a new handle and
    // validates two runs of the loaded plan. Every run uses the CPU reference of the original
    // graph.
    void roundTripInAFreshHandle(SaveForm form)
    {
        const auto cases = knobCases();
        const auto found = std::find_if(cases.begin(), cases.end(), [](const KnobCase& candidate) {
            return std::string(candidate.name) == SAVE_LOAD_CASE;
        });
        ASSERT_NE(found, cases.end()) << "no knob case is named " << SAVE_LOAD_CASE;
        ASSERT_TRUE(found->forced);
        const KnobCase& knobCase = *found;

        auto sdpa = detail::buildSdpaGraph(knobCase.name, knobCase.shape);
        {
            const ScopedPluginLogCapture capture(this);
            ASSERT_NO_FATAL_FAILURE(this->buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

            auto result
                = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(knobCase.tile));
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
            result = sdpa.graph->check_support();
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
            result = sdpa.graph->build_plans();
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

            // The load below must log no selection. This check proves that the capture
            // records selection lines.
            ASSERT_TRUE(capture.recorder().hasLogContaining(detail::SELECTED_KERNEL_LOG))
                << "Captured logs:\n"
                << capture.recorder().getRecordedLogsAsString();
        }

        int64_t servingEngineId = 0;
        auto result = sdpa.graph->get_execution_plan_engine_id(servingEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(servingEngineId, engineId());

        std::vector<BehaviorNote> notes;
        result = sdpa.graph->get_behavior_notes_for_engine(engineId(), notes);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(notes.begin(),
                            notes.end(),
                            BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION),
                  notes.end());

        VerificationRun run;
        {
            GraphVerificationContext context(*sdpa.graph);
            const float tolerance = detail::sdpaForwardTolerance(knobCase.shape.dataType);
            ASSERT_NO_FATAL_FAILURE(
                this->registerValidator(context, sdpa.output, tolerance, tolerance));
            ASSERT_NO_FATAL_FAILURE(this->prepareVerification(context, /*seed=*/0, run));
        }
        ASSERT_NO_FATAL_FAILURE(this->executeAndValidate(run, *sdpa.graph, this->_handle));

        std::vector<uint8_t> saved;
        {
            const ScopedPluginLogCapture capture(this);
            auto save = detail::saveInForm(*sdpa.graph, form);
            ASSERT_EQ(save.second.code, ErrorCode::OK) << save.second.err_msg;
            saved = std::move(save.first);
            EXPECT_FALSE(capture.recorder().hasLogContaining(
                "does not support execution plan serialization"))
                << "Captured logs:\n"
                << capture.recorder().getRecordedLogsAsString();
        }
        if(form == SaveForm::GRAPH_AND_PLAN)
        {
            int contents = 0;
            ASSERT_EQ(
                hipdnnBackendGetSerializedBinaryContents_ext(saved.data(), saved.size(), &contents),
                HIPDNN_STATUS_SUCCESS);
            ASSERT_NE(contents & HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN, 0)
                << "the saved graph holds no execution plan";
        }

        int64_t originalWorkspaceSize = 0;
        result = sdpa.graph->get_workspace_size(originalWorkspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const std::weak_ptr<Graph> original = sdpa.graph;
        sdpa = SdpaGraph{};
        ASSERT_TRUE(original.expired()) << "another owner keeps the original graph alive";

        const ScopedPluginLogCapture capture(this);
        ASSERT_NO_FATAL_FAILURE(detail::replaceHandleWithAFreshOne(this->_handle, this->_stream));

        auto loaded = std::make_shared<Graph>();
        result = detail::loadInForm(*loaded, this->_handle, saved, form);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        EXPECT_TRUE(capture.recorder().hasLogContaining(detail::CREATING_CONTAINER_LOG))
            << "the new handle did not get a new provider container. Captured logs:\n"
            << capture.recorder().getRecordedLogsAsString();
        detail::expectNoKernelSelectionLogged(capture.recorder());

        int64_t loadedEngineId = 0;
        result = loaded->get_execution_plan_engine_id(loadedEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(loadedEngineId, engineId());

        int64_t loadedWorkspaceSize = 0;
        result = loaded->get_workspace_size(loadedWorkspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(loadedWorkspaceSize, originalWorkspaceSize);

        ASSERT_NO_FATAL_FAILURE(this->executeAndValidate(run, *loaded, this->_handle));
        ASSERT_NO_FATAL_FAILURE(this->executeAndValidate(run, *loaded, this->_handle));
    }
};

TEST_F(IntegrationGpuGfx950AttentionDenseSaveLoad, RestoresGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN);
}

TEST_F(IntegrationGpuGfx950AttentionDenseSaveLoad, RestoresPlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY);
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx950AttentionDenseKnobs,
                         ::testing::ValuesIn(knobCases()),
                         [](const ::testing::TestParamInfo<KnobCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx950AttentionDenseKnobFilter,
                         ::testing::ValuesIn(unsatisfiableCases()),
                         [](const ::testing::TestParamInfo<UnsatisfiableCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
